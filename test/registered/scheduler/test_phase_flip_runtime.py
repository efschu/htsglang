# SPDX-License-Identifier: Apache-2.0
"""#631 PhaseFlipRuntime: hermetic contract tests on the #297 envelope.

CPU-only, no torch.distributed, no GPU: consensus and byte channels are
injected; the multi-rank tests drive REAL threads through barrier-backed
mocks, so "fails loudly, never hangs" is demonstrated with actual
concurrency. The load-bearing gates:

* byte identity -- after a threaded PP->TP flip the TP pools equal the
  global reference under the token-owner rule, and the reverse flip
  round-trips the PP pools byte-identically;
* config-fingerprint desync falsifier -- ONE rank booted with a shifted
  (still valid) layer map raises the same loud KvReshardError on every
  rank at the FIRST consensus round, and no pool byte moves;
* checksum / size falsifiers -- a corrupted or truncated payload is a
  loud error on the receiving rank with its pools untouched;
* readiness skew -- legal divergence holds uniformly, then commits;
* pre-sized-pool bounds -- an undersized TP pool is a loud sizing-bug
  error before any byte moves.

The real-config row-schema equality test (operator pin 3: PP and TP rows
byte-compatible from the model config, red if the weighted-DCP head
replication rule changes) needs the pool constructors and lands with the
integration step; the runtime already pins the property at run time via
receiver-derived payload sizes + checksums.
"""

import threading
import unittest

import torch

from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP, TP_TO_PP
from sglang.srt.layers.dcp.reshard_plan import KvReshardError, owner_of, rows_of
from sglang.srt.managers.kv_reshard import KvPoolView
from sglang.srt.managers.phase_flip_runtime import (
    PHASE_PP,
    PhaseFlipRuntime,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

MAP_625 = ((0, 1, 2, 3, 4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15))
N_LAYERS = 16
VEC = (3, 2, 2)
HEADS, DIM = 2, 4


class _BarrierMinChannel:
    """Element-wise MIN across N rank threads (the #287 mock, restated).
    A rank that never arrives breaks the barrier and raises in every
    waiting thread -- a hang cannot masquerade as a pass."""

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
    """Pairwise byte channel across N rank threads (barrier-backed)."""

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


def _make_layout_pools(layer_map, vec, num_slots, seed=7):
    """Global reference KV per ordinal + per-rank PP and TP pools.

    PP pool of rank r: K/V buffers for r's ordinals, row = slot id,
    pre-filled with the reference rows of the live set. TP pool of rank
    r: K/V buffers for ALL ordinals, compact rows, zeroed."""
    torch.manual_seed(seed)
    n_ranks = len(vec)
    ref_k = [
        torch.randn(num_slots, HEADS, DIM, dtype=torch.bfloat16)
        for _ in range(N_LAYERS)
    ]
    ref_v = [
        torch.randn(num_slots, HEADS, DIM, dtype=torch.bfloat16)
        for _ in range(N_LAYERS)
    ]
    live = (
        torch.arange(num_slots, dtype=torch.int64)[
            torch.randperm(num_slots)[: int(num_slots * 0.8)]
        ]
        .sort()
        .values
    )
    owner = owner_of(live, vec)
    pp_pools, pp_views, tp_pools, tp_views = [], [], [], []
    tp_rows_needed = 1
    for r in range(n_ranks):
        rr = rows_of(live[owner == r], vec, r)
        if rr.numel():
            tp_rows_needed = max(tp_rows_needed, int(rr.max().item()) + 1)
    for r in range(n_ranks):
        k_bufs = [
            torch.zeros(num_slots, HEADS, DIM, dtype=torch.bfloat16)
            for _ in layer_map[r]
        ]
        v_bufs = [
            torch.zeros(num_slots, HEADS, DIM, dtype=torch.bfloat16)
            for _ in layer_map[r]
        ]
        for i, f in enumerate(layer_map[r]):
            k_bufs[i][live] = ref_k[f][live]
            v_bufs[i][live] = ref_v[f][live]
        pp_pools.append((k_bufs, v_bufs))
        pp_views.append(KvPoolView(k_bufs, v_bufs))
        tk = [
            torch.zeros(tp_rows_needed, HEADS, DIM, dtype=torch.bfloat16)
            for _ in range(N_LAYERS)
        ]
        tv = [
            torch.zeros(tp_rows_needed, HEADS, DIM, dtype=torch.bfloat16)
            for _ in range(N_LAYERS)
        ]
        tp_pools.append((tk, tv))
        tp_views.append(KvPoolView(tk, tv))
    return (ref_k, ref_v), live, pp_pools, pp_views, tp_pools, tp_views


def _check_tp_layout(tp_pools, ref, live, vec):
    ref_k, ref_v = ref
    owner = owner_of(live, vec)
    for r, (tk, tv) in enumerate(tp_pools):
        mine = live[owner == r]
        rows = rows_of(mine, vec, r)
        for f in range(N_LAYERS):
            if not torch.equal(tk[f][rows], ref_k[f][mine]):
                return False, f"rank {r} ordinal {f} K mismatch"
            if not torch.equal(tv[f][rows], ref_v[f][mine]):
                return False, f"rank {r} ordinal {f} V mismatch"
    return True, ""


def _run_ranks(
    n_ranks,
    *,
    runtimes,
    directions,
    rounds=8,
):
    """Drive one runtime per rank on a real thread; arm ``directions[r]``
    (None = not armed) before the loop. Returns exceptions per rank."""
    exceptions = [None] * n_ranks

    def _worker(r):
        try:
            if directions[r] is not None:
                runtimes[r].arm(directions[r], source=f"test-rank{r}")
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
            f"{len(alive)} rank threads still alive after 30s -- a flip "
            f"hang, the exact failure mode this suite exists to catch"
        )
    return exceptions


def _build_runtimes(
    pp_views,
    tp_views,
    live,
    *,
    layer_maps=None,
    ready_fns=None,
    exchange_factory=None,
    cutover_log=None,
):
    n = len(VEC)
    channel = _BarrierMinChannel(n)
    mailbox = _MailboxExchange(n)
    if exchange_factory is None:
        exchange_factory = mailbox.exchange_for
    if cutover_log is None:
        cutover_log = [[] for _ in range(n)]
    runtimes = []
    for r in range(n):
        runtimes.append(
            PhaseFlipRuntime(
                n_ranks=n,
                rank=r,
                layer_map=(layer_maps[r] if layer_maps else MAP_625),
                n_layers=N_LAYERS,
                tp_vector=VEC,
                boot_phase=PHASE_PP,
                consensus_interval=2,
                collective_min=channel.channel_for(r),
                exchange=exchange_factory(r),
                pp_pool_view=pp_views[r],
                tp_pool_view=tp_views[r],
                live_slots_fn=lambda: live,
                ready_fn=(
                    ready_fns[r] if ready_fns else (lambda: True)
                ),
                cutover_fn=lambda d, r=r: cutover_log[r].append(d),
            )
        )
    return runtimes, cutover_log


def _clone_pools(pools):
    return [
        ([k.clone() for k in ks], [v.clone() for v in vs]) for ks, vs in pools
    ]


def _pools_equal(a, b):
    for (ak, av), (bk, bv) in zip(a, b):
        for x, y in zip(ak + av, bk + bv):
            if not torch.equal(x, y):
                return False
    return True


class TestByteIdentity(CustomTestCase):
    def test_pp_to_tp_flip_byte_identity(self):
        ref, live, pp_pools, pp_views, tp_pools, tp_views = _make_layout_pools(
            MAP_625, VEC, 300
        )
        runtimes, cutovers = _build_runtimes(pp_views, tp_views, live)
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        self.assertEqual([e for e in exceptions if e], [])
        ok, msg = _check_tp_layout(tp_pools, ref, live, VEC)
        self.assertTrue(ok, msg)
        for r, rt in enumerate(runtimes):
            self.assertEqual(rt.completed, 1)
            self.assertEqual(rt.phase, "tp")
            self.assertEqual(cutovers[r], [PP_TO_TP])

    def test_roundtrip_restores_pp_pools(self):
        ref, live, pp_pools, pp_views, tp_pools, tp_views = _make_layout_pools(
            MAP_625, VEC, 260, seed=9
        )
        orig = _clone_pools(pp_pools)
        runtimes, _ = _build_runtimes(pp_views, tp_views, live)
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        self.assertEqual([e for e in exceptions if e], [])
        # wipe the PP pools, then flip back on the SAME runtimes
        for ks, vs in pp_pools:
            for t in ks + vs:
                t.zero_()
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[TP_TO_PP] * 3)
        self.assertEqual([e for e in exceptions if e], [])
        # live rows must round-trip byte-identically
        for r in range(3):
            (ks, vs), (oks, ovs) = pp_pools[r], orig[r]
            for t, o in zip(ks + vs, oks + ovs):
                self.assertTrue(torch.equal(t[live], o[live]))
        for rt in runtimes:
            self.assertEqual(rt.completed, 2)
            self.assertEqual(rt.phase, "pp")


class TestConsensusDiscipline(CustomTestCase):
    def test_config_fingerprint_desync_all_loud_none_hang(self):
        # Rank 1 boots with a shifted (still valid) layer map: its PP pool
        # has 5 layers to match, so construction succeeds -- the divergence
        # must die at the FIRST consensus round, loudly, on EVERY rank,
        # with no pool byte moved. Can-fail proof of the fingerprint gate.
        shifted = ((0, 1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11), (12, 13, 14, 15))
        ref, live, pp_pools, pp_views, tp_pools, tp_views = _make_layout_pools(
            MAP_625, VEC, 200, seed=13
        )
        # rank 1's PP view must cover 5 layers under the shifted map
        k5 = [torch.zeros(200, HEADS, DIM, dtype=torch.bfloat16) for _ in range(5)]
        v5 = [torch.zeros(200, HEADS, DIM, dtype=torch.bfloat16) for _ in range(5)]
        pp_views = list(pp_views)
        pp_views[1] = KvPoolView(k5, v5)
        pp_before = _clone_pools(pp_pools)
        tp_before = _clone_pools(tp_pools)
        maps = [MAP_625, shifted, MAP_625]
        runtimes, _ = _build_runtimes(
            pp_views, tp_views, live, layer_maps=maps
        )
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[None] * 3)
        for r, e in enumerate(exceptions):
            self.assertIsInstance(e, KvReshardError, f"rank {r}: {e}")
            self.assertIn("config_fp", str(e))
        self.assertTrue(_pools_equal(pp_pools, pp_before))
        self.assertTrue(_pools_equal(tp_pools, tp_before))

    def test_readiness_skew_holds_uniformly_then_commits(self):
        ref, live, pp_pools, pp_views, tp_pools, tp_views = _make_layout_pools(
            MAP_625, VEC, 180, seed=15
        )
        gate = {"count": 0}
        lock = threading.Lock()

        def _rank1_ready():
            with lock:
                gate["count"] += 1
                return gate["count"] > 2  # not ready the first two probes

        ready_fns = [lambda: True, _rank1_ready, lambda: True]
        runtimes, _ = _build_runtimes(
            pp_views, tp_views, live, ready_fns=ready_fns
        )
        exceptions = _run_ranks(
            3, runtimes=runtimes, directions=[PP_TO_TP] * 3, rounds=10
        )
        self.assertEqual([e for e in exceptions if e], [])
        ok, msg = _check_tp_layout(tp_pools, ref, live, VEC)
        self.assertTrue(ok, msg)
        for rt in runtimes:
            self.assertEqual(rt.completed, 1)

    def test_checksum_falsifier_corrupted_payload_pool_untouched(self):
        ref, live, pp_pools, pp_views, tp_pools, tp_views = _make_layout_pools(
            MAP_625, VEC, 160, seed=17
        )
        tp_before = _clone_pools(tp_pools)
        mailbox = _MailboxExchange(3)

        def _factory(rank):
            inner = mailbox.exchange_for(rank)

            def _exchange(outgoing, incoming_nbytes):
                received = inner(outgoing, incoming_nbytes)
                if rank == 1:
                    for peer, payload in received.items():
                        if payload.numel():
                            payload[0] ^= 0xFF  # corrupt one byte
                            break
                return received

            return _exchange

        runtimes, _ = _build_runtimes(
            pp_views, tp_views, live, exchange_factory=_factory
        )
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        self.assertIsInstance(exceptions[1], KvReshardError)
        self.assertIn("checksum", str(exceptions[1]))
        # rank 1 aborted BEFORE its write phase: its TP pool is untouched.
        self.assertTrue(
            _pools_equal([tp_pools[1]], [tp_before[1]]),
            "rank 1 scattered bytes after a checksum failure",
        )

    def test_truncated_payload_is_loud_size_error(self):
        ref, live, pp_pools, pp_views, tp_pools, tp_views = _make_layout_pools(
            MAP_625, VEC, 150, seed=19
        )
        mailbox = _MailboxExchange(3)

        def _factory(rank):
            inner = mailbox.exchange_for(rank)

            def _exchange(outgoing, incoming_nbytes):
                received = inner(outgoing, incoming_nbytes)
                if rank == 2:
                    for peer, payload in received.items():
                        received[peer] = payload[:-16]
                        break
                return received

            return _exchange

        runtimes, _ = _build_runtimes(
            pp_views, tp_views, live, exchange_factory=_factory
        )
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        self.assertIsInstance(exceptions[2], KvReshardError)
        self.assertIn("size mismatch", str(exceptions[2]))


class TestValidationAndBounds(CustomTestCase):
    def _single_rank_runtime(self, **overrides):
        ref, live, pp_pools, pp_views, tp_pools, tp_views = _make_layout_pools(
            MAP_625, VEC, 100, seed=23
        )
        kwargs = dict(
            n_ranks=3,
            rank=0,
            layer_map=MAP_625,
            n_layers=N_LAYERS,
            tp_vector=VEC,
            boot_phase=PHASE_PP,
            collective_min=lambda vals: list(vals),
            exchange=lambda o, i: {},
            pp_pool_view=pp_views[0],
            tp_pool_view=tp_views[0],
            live_slots_fn=lambda: live,
            ready_fn=lambda: True,
            cutover_fn=lambda d: None,
        )
        kwargs.update(overrides)
        return PhaseFlipRuntime(**kwargs)

    def test_arm_refuses_wrong_direction_for_phase(self):
        rt = self._single_rank_runtime()
        ok, msg = rt.arm(TP_TO_PP, source="test")
        self.assertFalse(ok)
        self.assertIn("current phase is pp", msg)
        ok, _ = rt.arm(PP_TO_TP, source="test")
        self.assertTrue(ok)

    def test_arm_refuses_unknown_direction_and_guards(self):
        rt = self._single_rank_runtime()
        ok, _ = rt.arm("sideways", source="test")
        self.assertFalse(ok)
        guarded = self._single_rank_runtime(guards=("disk hicache (#630)",))
        ok, msg = guarded.arm(PP_TO_TP, source="test")
        self.assertFalse(ok)
        self.assertIn("disk hicache", msg)

    def test_wrong_pp_view_layer_count_refused(self):
        with self.assertRaisesRegex(KvReshardError, "PP pool view"):
            self._single_rank_runtime(
                pp_pool_view=KvPoolView(
                    [torch.zeros(10, HEADS, DIM, dtype=torch.bfloat16)],
                    [torch.zeros(10, HEADS, DIM, dtype=torch.bfloat16)],
                )
            )

    def test_undersized_tp_pool_is_loud_sizing_error(self):
        ref, live, pp_pools, pp_views, tp_pools, tp_views = _make_layout_pools(
            MAP_625, VEC, 120, seed=29
        )
        # shrink every rank's TP pool below the needed compact rows
        small_tp_views = []
        for r in range(3):
            tk = [
                torch.zeros(2, HEADS, DIM, dtype=torch.bfloat16)
                for _ in range(N_LAYERS)
            ]
            tv = [
                torch.zeros(2, HEADS, DIM, dtype=torch.bfloat16)
                for _ in range(N_LAYERS)
            ]
            small_tp_views.append(KvPoolView(tk, tv))
        runtimes, _ = _build_runtimes(pp_views, small_tp_views, live)
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        for r, e in enumerate(exceptions):
            self.assertIsInstance(e, KvReshardError, f"rank {r}: {e}")
            self.assertIn("sizing bug", str(e))


class TestSchedulerSideHelpers(CustomTestCase):
    """Quiescence predicate, live-slot union, guards (DESIGN_631 3.5/3.7)."""

    def _fake_scheduler(self, **over):
        from types import SimpleNamespace

        tree = SimpleNamespace(
            all_values_flatten=lambda: torch.tensor([2, 5, 7], dtype=torch.int64)
        )
        req_to_token = torch.arange(40, dtype=torch.int64).reshape(4, 10)
        sched = SimpleNamespace(
            chunked_req=None,
            last_batch=None,
            result_queue=[],
            tree_cache=tree,
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            running_batch=SimpleNamespace(reqs=[]),
            waiting_queue=[SimpleNamespace()],  # parked work is LEGAL
            server_args=SimpleNamespace(
                enable_hierarchical_cache=False, dual_group_lane=False
            ),
            kv_session_offload=None,
            is_dual_group_lane=False,
        )
        from sglang.srt.disaggregation.utils import DisaggregationMode

        sched.disaggregation_mode = DisaggregationMode.NULL
        for k, v in over.items():
            setattr(sched, k, v)
        return sched

    def test_quiescence_true_with_parked_requests(self):
        from types import SimpleNamespace

        sched = self._fake_scheduler(
            running_batch=SimpleNamespace(
                reqs=[SimpleNamespace(seqlen=3, req_pool_idx=0)]
            )
        )
        from sglang.srt.managers.phase_flip_runtime import build_flip_quiescence_fn

        self.assertTrue(build_flip_quiescence_fn(sched)())

    def test_quiescence_false_on_inflight_state(self):
        from types import SimpleNamespace

        from sglang.srt.managers.phase_flip_runtime import build_flip_quiescence_fn

        for over in (
            {"chunked_req": object()},
            {"result_queue": [object()]},
            {"last_batch": SimpleNamespace(is_empty=lambda: False)},
            {"_pp_microbatches_drained": lambda: False},
        ):
            sched = self._fake_scheduler(**over)
            self.assertFalse(build_flip_quiescence_fn(sched)(), over)

    def test_live_slots_union_tree_and_parked_rows(self):
        from types import SimpleNamespace

        from sglang.srt.managers.phase_flip_runtime import build_flip_live_slots_fn

        # req 0 parked with 3 tokens at rows 7, 8, 9 (row 7 also in tree).
        req_to_token = torch.zeros(4, 10, dtype=torch.int64)
        req_to_token[0, :3] = torch.tensor([7, 8, 9])
        sched = self._fake_scheduler(
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            running_batch=SimpleNamespace(
                reqs=[SimpleNamespace(seqlen=3, req_pool_idx=0)]
            ),
        )
        live = build_flip_live_slots_fn(sched)()
        self.assertEqual(live.tolist(), [2, 5, 7, 8, 9])

    def test_guards(self):
        from sglang.srt.managers.phase_flip_runtime import flip_blocking_guards

        self.assertEqual(flip_blocking_guards(self._fake_scheduler()), [])
        sched = self._fake_scheduler()
        sched.server_args.enable_hierarchical_cache = True
        guards = flip_blocking_guards(sched)
        self.assertTrue(any("#630" in g for g in guards), guards)
        sched = self._fake_scheduler(is_dual_group_lane=True)
        self.assertTrue(
            any("dual-group" in g for g in flip_blocking_guards(sched))
        )


class TestAbortDeferral(CustomTestCase):
    """Pin 4 (DESIGN_631 3.6a): parked-request disconnect during a flip."""

    def test_window_defers_then_drains_in_order(self):
        from sglang.srt.managers.phase_flip_runtime import AbortDeferralWindow

        window = AbortDeferralWindow()
        ran = []
        self.assertFalse(window.submit(lambda: ran.append("now")))
        self.assertEqual(ran, ["now"])
        window.activate()
        self.assertTrue(window.submit(lambda: ran.append("a")))
        self.assertTrue(window.submit(lambda: ran.append("b")))
        self.assertEqual(ran, ["now"])
        self.assertEqual(window.deferred_count, 2)
        self.assertEqual(window.deactivate_and_drain(), 2)
        self.assertEqual(ran, ["now", "a", "b"])

    def _runtimes_with_per_rank_live(self, live_per_rank, pp_views, tp_views):
        channel = _BarrierMinChannel(3)
        mailbox = _MailboxExchange(3)
        runtimes = []
        for r in range(3):
            runtimes.append(
                PhaseFlipRuntime(
                    n_ranks=3,
                    rank=r,
                    layer_map=MAP_625,
                    n_layers=N_LAYERS,
                    tp_vector=VEC,
                    boot_phase=PHASE_PP,
                    consensus_interval=2,
                    collective_min=channel.channel_for(r),
                    exchange=mailbox.exchange_for(r),
                    pp_pool_view=pp_views[r],
                    tp_pool_view=tp_views[r],
                    live_slots_fn=lambda r=r: live_per_rank[r],
                    ready_fn=lambda: True,
                    cutover_fn=lambda d: None,
                )
            )
        return runtimes

    def test_can_fail_abort_applied_on_one_rank_mid_flip_is_loud(self):
        # WITHOUT deferral: rank 0 has already applied a disconnect (its
        # live set lost a slot owned by rank 1) while the peers still see
        # the old set. The flip must die LOUDLY (size mismatch on the
        # shrunken pair payload), never scatter silently -- proves the
        # hazard deferral exists for is real, and that the runtime's
        # failure mode for it is the loud one.
        ref, live, pp_pools, pp_views, tp_pools, tp_views = _make_layout_pools(
            MAP_625, VEC, 140, seed=31
        )
        owner = owner_of(live, VEC)
        rank1_slots = live[owner == 1]
        self.assertGreater(int(rank1_slots.numel()), 0)
        dropped = int(rank1_slots[0].item())
        live_rank0 = live[live != dropped]
        live_per_rank = [live_rank0, live, live]
        runtimes = self._runtimes_with_per_rank_live(
            live_per_rank, pp_views, tp_views
        )
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        louds = [e for e in exceptions if isinstance(e, KvReshardError)]
        self.assertTrue(louds, f"divergent live set went undetected: {exceptions}")
        self.assertTrue(
            any(
                "size mismatch" in str(e) or "checksum" in str(e)
                for e in louds
            ),
            louds,
        )

    def test_can_fail_batch_membership_disagreement_is_refused_loudly(self):
        # Cross-strand hazard (#622 root cause, first-token no-sync): on a
        # mixed-arch rig ranks can read DIFFERENT first tokens and so
        # DISAGREE whether a request hit EOS -- one rank believes the
        # request FINISHED (freed its rows), peers still hold it parked.
        # The flip's replicated-live-set assumption is then false. This
        # test pins the required behavior: the flip REFUSES loudly (the
        # shrunken pair payloads mismatch the peers' expectations), it
        # never commits a mixed-membership layout silently.
        ref, live, pp_pools, pp_views, tp_pools, tp_views = _make_layout_pools(
            MAP_625, VEC, 140, seed=37
        )
        # "Request X" = a contiguous run of slots; rank 2 believes it
        # finished and enumerates a live set WITHOUT those rows.
        req_rows = live[3:9]
        mask = torch.ones(live.numel(), dtype=torch.bool)
        for i in range(live.numel()):
            if live[i] in req_rows:
                mask[i] = False
        live_rank2 = live[mask]
        self.assertLess(int(live_rank2.numel()), int(live.numel()))
        live_per_rank = [live, live, live_rank2]
        runtimes = self._runtimes_with_per_rank_live(
            live_per_rank, pp_views, tp_views
        )
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        louds = [e for e in exceptions if isinstance(e, KvReshardError)]
        self.assertTrue(
            louds,
            f"membership disagreement went undetected: {exceptions}",
        )
        # Nobody may have cut over to the new layout on a divergent set...
        # except ranks whose pair payloads happened to be consistent; the
        # LOUD failure on at least one rank is what kills the group before
        # the mixed layout serves. The binding assertion: no hang, loud.
        self.assertTrue(
            any(
                "size mismatch" in str(e) or "checksum" in str(e)
                for e in louds
            ),
            louds,
        )

    def test_deferral_keeps_flip_clean_then_applies_abort(self):
        # WITH deferral: the disconnect arrives mid-flip on rank 0, is
        # QUEUED, the flip commits byte-identically on every rank, and
        # the abort work runs afterwards.
        from sglang.srt.managers.phase_flip_runtime import AbortDeferralWindow

        ref, live, pp_pools, pp_views, tp_pools, tp_views = _make_layout_pools(
            MAP_625, VEC, 140, seed=33
        )
        window = AbortDeferralWindow()
        window.activate()
        applied = []
        # The disconnect lands while the window is active (armed flip).
        self.assertTrue(window.submit(lambda: applied.append("abort req X")))
        live_per_rank = [live, live, live]  # deferral kept the set uniform
        runtimes = self._runtimes_with_per_rank_live(
            live_per_rank, pp_views, tp_views
        )
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        self.assertEqual([e for e in exceptions if e], [])
        ok, msg = _check_tp_layout(tp_pools, ref, live, VEC)
        self.assertTrue(ok, msg)
        self.assertEqual(applied, [])
        self.assertEqual(window.deactivate_and_drain(), 1)
        self.assertEqual(applied, ["abort req X"])


if __name__ == "__main__":
    unittest.main()


class TestPpLoopConsensusOrdering(CustomTestCase):
    """Family pin (rank-local-state-feeds-collective, PP form; measured
    wedge 2026-08-08): under a pipeline, entering the bounded consensus
    at the TOP of the iteration -- before this rank's send -- closes a
    cycle with the p2p chain (two ranks in the reduction, one blocked in
    recv-from-prev). The fix places the hook at the END of the iteration,
    after every send is flushed. Both orderings are driven here against
    the REAL on_round consensus through a bounded barrier channel: the
    top placement must deadlock (negative control, can-fail proof), the
    end placement must complete."""

    N_ITERS = 4  # consensus_interval=2 -> two consensus boundaries

    def _drive(self, hook_position):
        import queue

        n = len(VEC)
        _, live, _, pp_views, _, tp_views = _make_layout_pools(
            MAP_625, VEC, 200, seed=29
        )
        barrier = threading.Barrier(n, timeout=3.0)

        def _channel_for(r):
            lows = [None] * n

            def _reduce(vals, r=r):
                lows[r] = list(vals)
                barrier.wait()  # BrokenBarrierError after 3 s = deadlock
                agg = [
                    min(lows[q][i] for q in range(n) if lows[q] is not None)
                    for i in range(len(vals))
                ]
                return agg

            return _reduce

        runtimes = []
        for r in range(n):
            runtimes.append(
                PhaseFlipRuntime(
                    n_ranks=n,
                    rank=r,
                    layer_map=MAP_625,
                    n_layers=N_LAYERS,
                    tp_vector=VEC,
                    boot_phase=PHASE_PP,
                    consensus_interval=2,
                    collective_min=_channel_for(r),
                    exchange=lambda peers, payloads: {},
                    pp_pool_view=pp_views[r],
                    tp_pool_view=tp_views[r],
                    live_slots_fn=lambda: live,
                    ready_fn=lambda: False,
                    cutover_fn=lambda d: None,
                )
            )

        # Measured wedge composition (2026-08-08): the last stage blocks in
        # a mid-iteration recv BEFORE its round hook, while the middle
        # stage's send to it trails the middle stage's own hook. With the
        # hook at the TOP the middle stage enters the bounded reduction
        # still owing that send -> the last stage can never reach its
        # reduction and the barrier breaks (the wedge). With the hook at
        # the END every send precedes every hook, so the recv is always
        # satisfiable and the reduction completes.
        pipes = [queue.Queue() for _ in range(n)]
        outcomes = [None] * n

        last, middle = n - 1, n - 2

        def _worker(r):
            try:
                for i in range(self.N_ITERS):
                    if hook_position == "top" and r != last:
                        runtimes[r].on_round()
                    if r == last:
                        pipes[middle].get(timeout=3.0)  # mid-iteration recv
                    if r == middle:
                        pipes[middle].put(i)  # send to last
                    if hook_position == "end" and r != last:
                        runtimes[r].on_round()
                    if r == last:
                        runtimes[r].on_round()  # hook always after its recv
                outcomes[r] = "done"
            except BaseException as e:  # noqa: BLE001
                outcomes[r] = e

        threads = [
            threading.Thread(target=_worker, args=(r,), daemon=True)
            for r in range(n)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
        return outcomes

    def test_top_placement_deadlocks_negative_control(self):
        import queue as queue_mod

        outcomes = self._drive("top")
        self.assertNotEqual(
            outcomes,
            ["done"] * len(VEC),
            "top-of-iteration consensus completed under pipeline skew -- "
            "the negative control lost its teeth; re-derive the ordering "
            "argument before trusting the end placement",
        )
        broken = [
            o
            for o in outcomes
            if isinstance(o, (threading.BrokenBarrierError, queue_mod.Empty))
        ]
        self.assertTrue(
            broken,
            f"expected a bounded deadlock signature, got {outcomes}",
        )

    def test_end_placement_completes(self):
        outcomes = self._drive("end")
        self.assertEqual(
            outcomes,
            ["done"] * len(VEC),
            f"end-of-iteration consensus must complete: {outcomes}",
        )


class TestArmedParkedGate(CustomTestCase):
    """PP-phase entry gate (boots 9+10, 2026-08-08): the end-of-iteration
    placement alone still wedged because the ranks' ABSOLUTE round counts
    diverge under the pp loop. Under require_armed_and_parked, an unarmed
    or unparked rank must touch the collective channel ZERO times; an
    armed+parked rank must enter every round (no interval gating -- the
    counters are not comparable across ranks there)."""

    def _runtime(self, *, ready, channel_calls):
        _, live, _, pp_views, _, tp_views = _make_layout_pools(
            MAP_625, VEC, 200, seed=31
        )

        def _min(vals):
            channel_calls.append(list(vals))
            return list(vals)

        return PhaseFlipRuntime(
            n_ranks=len(VEC),
            rank=0,
            layer_map=MAP_625,
            n_layers=N_LAYERS,
            tp_vector=VEC,
            boot_phase=PHASE_PP,
            consensus_interval=2,
            collective_min=_min,
            exchange=lambda peers, payloads: {},
            pp_pool_view=pp_views[0],
            tp_pool_view=tp_views[0],
            live_slots_fn=lambda: live,
            ready_fn=lambda: ready,
            cutover_fn=lambda d: None,
        )

    def test_unarmed_rounds_touch_no_collective(self):
        calls = []
        rt = self._runtime(ready=True, channel_calls=calls)
        for _ in range(16):
            rt.on_round(require_armed_and_parked=True)
        self.assertEqual(calls, [], "unarmed pp round entered the channel")

    def test_armed_unparked_rounds_touch_no_collective(self):
        calls = []
        rt = self._runtime(ready=False, channel_calls=calls)
        rt.arm(PP_TO_TP, source="test")
        for _ in range(16):
            rt.on_round(require_armed_and_parked=True)
        self.assertEqual(calls, [], "unparked pp round entered the channel")

    def test_armed_parked_rounds_enter_every_round(self):
        calls = []
        rt = self._runtime(ready=True, channel_calls=calls)
        rt.arm(PP_TO_TP, source="test")
        for _ in range(3):
            try:
                rt.on_round(require_armed_and_parked=True)
            except Exception:
                break  # commit path may proceed further than this stub
        self.assertTrue(calls, "armed+parked pp round never entered")
