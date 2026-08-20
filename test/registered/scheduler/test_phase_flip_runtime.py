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

from sglang.srt.layers.dcp.phase_flip_plan import (
    PP_TO_TP,
    TP_TO_PP,
    default_wave_count,
)
from sglang.srt.layers.dcp.reshard_plan import KvReshardError, owner_of, rows_of
from sglang.srt.managers.kv_reshard import KvPoolView
from sglang.srt.managers.phase_flip_runtime import (
    PHASE_PP,
    PHASE_TP,
    PhaseFlipRuntime,
    WavedBackingSwap,
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
    pre_write_fns=(),
    pre_write_fns_for=None,
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
                ready_fn=(ready_fns[r] if ready_fns else (lambda: True)),
                cutover_fn=lambda d, r=r: cutover_log[r].append(d),
                pre_write_fns=(
                    pre_write_fns_for(r) if pre_write_fns_for else pre_write_fns
                ),
            )
        )
    return runtimes, cutover_log


def _clone_pools(pools):
    return [([k.clone() for k in ks], [v.clone() for v in vs]) for ks, vs in pools]


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
        runtimes, _ = _build_runtimes(pp_views, tp_views, live, layer_maps=maps)
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
        runtimes, _ = _build_runtimes(pp_views, tp_views, live, ready_fns=ready_fns)
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

    def test_undersized_tp_pool_abandons_the_flip_without_killing_serving(self):
        """The bound is checked before any byte moves, so the answer to
        "it does not fit" is to abandon the FLIP, unanimously.

        This used to raise. Two things were wrong with that (#631 window
        3, boot 19, where it killed a server that was serving fine): the
        reading is RANK-LOCAL, so a rank that raised while a peer
        proceeded would leave the group half-flipped; and the raise
        climbed into the event loop and took the instance down with every
        request on it. Whether the live set fits is a RUNTIME quantity --
        it grows with the resident prefix cache, and the TP pool shrinks
        when a draft-KV allocation shares its budget -- so it is not a
        boot-time-only sizing bug that may abort the process.
        """
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
            self.assertIsNone(e, f"rank {r} raised instead of abandoning: {e}")
        for r, rt in enumerate(runtimes):
            self.assertEqual(rt.fit_aborts, 1, f"rank {r} did not abandon")
            self.assertEqual(rt.completed, 0, f"rank {r} flipped anyway")
            self.assertEqual(rt.epoch, 0, f"rank {r} advanced the epoch")
            self.assertIsNone(rt.pending, f"rank {r} stayed armed")
            self.assertEqual(rt.phase, PHASE_PP, f"rank {r} changed phase")
        # And the pools are untouched: the abandon happens before the plan
        # is executed, so this is a no-op, not a partial move.
        for r in range(3):
            for buf in small_tp_views[r]._k:
                self.assertEqual(float(buf.abs().sum()), 0.0, f"rank {r} wrote bytes")


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
            # A request that exists ONLY in last_batch: not yet merged
            # into the resident set the carry harvests, so flipping now
            # would leave it behind. Clears itself in one iteration.
            {"last_batch": SimpleNamespace(reqs=[SimpleNamespace(rid="new")])},
            # An in-flight PP microbatch: the pipeline is not quiet.
            {"mbs": [SimpleNamespace(is_empty=lambda: False)]},
        ):
            sched = self._fake_scheduler(**over)
            self.assertFalse(build_flip_quiescence_fn(sched)(), over)

    def test_last_batch_mirroring_the_resident_set_does_not_block_the_flip(self):
        """#631 DEFECT L, the regression arm.

        Under event_loop_normal (the TP decode phase) the result is
        processed in the SAME iteration as the forward, and last_batch is
        then set to that batch. So at the hook a non-empty last_batch
        means "requests are resident", not "work is in flight" -- and the
        old predicate refused on it, which made tp_to_pp unable to reach a
        quiescent boundary while anything was decoding. Measured on metal
        03:11:22Z and 03:12:52Z: "NOT QUIESCENT: last_batch is not empty
        (1 req(s) visible)" on all three ranks, abandoned at the park
        deadline both times, minutes after pp_to_tp had carried the very
        same request across the other way.

        The same category error as the _pp_microbatches_drained one: a
        term that refuses because requests EXIST.
        """
        from types import SimpleNamespace

        from sglang.srt.managers.phase_flip_runtime import build_flip_quiescence_fn

        req = SimpleNamespace(seqlen=3, req_pool_idx=0, rid="r0")
        running = SimpleNamespace(reqs=[req])
        sched = self._fake_scheduler(running_batch=running, last_batch=running)
        self.assertTrue(build_flip_quiescence_fn(sched)())
        # Same requests, a DIFFERENT batch object: still not an orphan.
        sched = self._fake_scheduler(
            running_batch=running, last_batch=SimpleNamespace(reqs=[req])
        )
        self.assertTrue(build_flip_quiescence_fn(sched)())

    def test_speculating_tp_phase_waits_for_the_resident_set_to_drain(self):
        """#631: a carried request has NO DRAFT STATE.

        It prefilled in the PP phase, which carries no draft worker by
        design, so nothing ran the draft_extend a spec instance gives a
        request after its target extend. Carrying it into a speculating TP
        phase killed the instance one pass later -- measured 03:32:14Z on
        all three ranks, "output with shape [1, 1] doesn't match the
        broadcast shape [0, 1]" inside the draft graph runner's
        foreach_copy, then SIGQUIT.

        So the rank is NOT READY while anything is resident -- waiting,
        not refusing at arm time: a rank-local refusal would let one rank
        decline while its peers armed, and diverging epochs is corpse H.
        """
        from types import SimpleNamespace

        from sglang.srt.managers.phase_flip_runtime import (
            PP_TO_TP,
            build_flip_quiescence_fn,
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        req = SimpleNamespace(seqlen=3, req_pool_idx=0, rid="r0")
        running = SimpleNamespace(reqs=[req])

        def _sched(algo, reqs):
            s = self._fake_scheduler(running_batch=SimpleNamespace(reqs=reqs))
            s.flip_spec_algorithm = SpeculativeAlgorithm.from_string(algo)
            s.phase_flip_runtime = SimpleNamespace(pending=PP_TO_TP)
            return s

        # Speculating TP phase + a resident request -> hold.
        sched = _sched("EAGLE", [req])
        self.assertFalse(build_flip_quiescence_fn(sched)())
        self.assertIn("no draft state", build_flip_quiescence_fn(sched).why_not())
        # Speculating, but nothing to carry -> the flip may go (this is
        # the regime every flip before the carry ran in).
        self.assertTrue(build_flip_quiescence_fn(_sched("EAGLE", []))())
        # No speculation -> the carry handles it; residents do not hold.
        self.assertTrue(build_flip_quiescence_fn(_sched(None, [req]))())

    def test_a_resident_decode_set_alone_does_not_block_the_flip(self):
        """THE CONTRACT, and the correction of a measured contradiction.

        This used to call Scheduler._pp_microbatches_drained, the
        FULLY-IDLE predicate, which also requires every ``running_mbs``
        slot to be empty. ``running_mbs`` is the RESIDENT DECODE SET: it
        empties only when the requests FINISH. So quiescence could not
        hold while anything was decoding -- and the automatic policy arms
        pp_to_tp precisely BECAUSE requests are decoding. The two
        conditions were mutually exclusive, and every automatic flip
        abandoned at the park deadline.

        Measured 2026-08-09 01:29:50Z under POLICY=auto with one request
        decoding: ranks 0 and 1 reported "PP microbatches not drained
        (live mb slots [], running_mbs slots [0])" -- nothing in flight,
        the resident decode set alone holding the flip. The gate
        assembled, all three ranks entered the reduction, and the group
        agreed to abandon with ready=0 everywhere.

        Carrying a resident decode set across is what the rest of the
        design already assumes: build_flip_live_slots_fn exists to move
        exactly those requests' KV rows.
        """
        from types import SimpleNamespace

        from sglang.srt.managers.phase_flip_runtime import build_flip_quiescence_fn

        sched = self._fake_scheduler(
            mbs=[None, None, None],
            running_mbs=[SimpleNamespace(is_empty=lambda: False)],
        )
        self.assertTrue(
            build_flip_quiescence_fn(sched)(),
            "a resident decode set blocked quiescence; the flip can then "
            "never commit under the policy that arms it",
        )

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
        # #703: the HiCache clause is REMOVED, in both of its copies. It was
        # first narrowed to "is the DISK tier configured" and then dropped
        # entirely, because the disk tier is the retention store the design
        # needs and the wedge behind the guard (#630) is fixed and shipped
        # (9da9dfd025 bounded collectives; see test_hicache_bounded_waits_630.py
        # and test/registered/unit/managers/test_hicache_flip_guard_703.py).
        # What stays gated is the KV key's pp suffix, which is a claim about
        # bytes and belongs with #706's whole-page format -- not this list.
        sched = self._fake_scheduler()
        sched.server_args.enable_hierarchical_cache = True
        self.assertEqual(flip_blocking_guards(sched), [])
        sched.server_args.hicache_storage_backend = "file"
        self.assertEqual(flip_blocking_guards(sched), [])
        sched = self._fake_scheduler(is_dual_group_lane=True)
        self.assertTrue(any("dual-group" in g for g in flip_blocking_guards(sched)))


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

    def test_can_fail_abort_applied_on_one_rank_mid_flip_is_refused(self):
        # WITHOUT deferral: rank 0 has already applied a disconnect (its
        # live set lost a slot owned by rank 1) while the peers still see
        # the old set. The flip must REFUSE, never scatter silently --
        # proves the hazard deferral exists for is real.
        #
        # #656 C22: this used to assert a loud RAISE, because a size or
        # checksum error on the shrunken pair payload was the only thing
        # that noticed. Raising at the seam is what killed the acceptance
        # instance, and on the real transport the divergence does not
        # even reach a size error -- NCCL delivers the short count into a
        # receiver-allocated buffer and the trailer read from its
        # unwritten tail is not a checksum at all. The pre-move frame
        # ballot now catches the SAME divergence one collective earlier,
        # so the assertions here get strictly stronger: not merely "some
        # rank died before the mixed layout served", but "no byte moved
        # on any rank". The can-fail arm proving this can still go red
        # lives in test_flip_frame_agreement_656.py, which stubs the
        # digest out and recovers the metal signature.
        ref, live, pp_pools, pp_views, tp_pools, tp_views = _make_layout_pools(
            MAP_625, VEC, 140, seed=31
        )
        owner = owner_of(live, VEC)
        rank1_slots = live[owner == 1]
        self.assertGreater(int(rank1_slots.numel()), 0)
        dropped = int(rank1_slots[0].item())
        live_rank0 = live[live != dropped]
        #
        # #656 C22-d: the live-slot AGREEMENT now repairs this class before
        # the ballot votes, so the shipped behaviour is no longer an abandon.
        # This arm keeps measuring the hazard by DISARMING the agreement --
        # which is what makes it a can-fail arm at all: it has to reproduce
        # the state the protection is absent in. The repaired behaviour is
        # pinned by its own test below, so both halves are measured and
        # neither is assumed.
        live_per_rank = [live_rank0, live, live]
        tp_before = _clone_pools(tp_pools)
        runtimes = self._runtimes_with_per_rank_live(live_per_rank, pp_views, tp_views)
        for rt in runtimes:
            rt._agree_live_slots = lambda slots, ballot: (slots, "")
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        self.assertEqual([e for e in exceptions if e is not None], [])
        for r, rt in enumerate(runtimes):
            self.assertEqual(rt.frame_aborts, 1, f"rank {r}")
            self.assertEqual(rt.completed, 0, f"rank {r}")
        self.assertTrue(
            _pools_equal(tp_pools, tp_before),
            "a divergent live set scattered bytes into the TP layout",
        )

    def test_the_agreement_repairs_the_dropped_slot_instead_of_wedging(self):
        """#656 C22-d, the shipped half of the arm above.

        The same divergence, the agreement ARMED. The group must agree on
        the union and cut over -- because the alternative, measured on
        boot ``boot_m3``, is that the ranks re-frame the same disagreement
        every round and the pp_to_tp leg (the one decode needs) is never
        taken again. The disagreement about WHOSE request that row belongs
        to is a real and separate defect; it is counted and logged here,
        not silently absorbed, and the flip no longer wedges on it.
        """
        ref, live, pp_pools, pp_views, tp_pools, tp_views = _make_layout_pools(
            MAP_625, VEC, 140, seed=31
        )
        owner = owner_of(live, VEC)
        rank1_slots = live[owner == 1]
        dropped = int(rank1_slots[0].item())
        live_rank0 = live[live != dropped]
        runtimes = self._runtimes_with_per_rank_live(
            [live_rank0, live, live], pp_views, tp_views
        )
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        self.assertEqual([e for e in exceptions if e is not None], [])
        for r, rt in enumerate(runtimes):
            self.assertEqual(rt.frame_aborts, 0, f"rank {r}")
            self.assertEqual(rt.completed, 1, f"rank {r}")
            self.assertEqual(
                (rt.slot_set_divergences, rt.slot_set_agreements),
                (1, 1),
                f"rank {r}: the flip went through WITHOUT the agreement "
                f"firing, which would mean a divergence was never seen -- "
                f"silence, not repair",
            )
            framed = set(int(x) for x in rt.last_framed_slots.tolist())
            self.assertIn(
                dropped,
                framed,
                f"rank {r} framed a set missing the row two of its peers "
                f"still hold -- that row's KV would be lost at the seam",
            )

    def test_can_fail_batch_membership_disagreement_is_refused(self):
        # Cross-strand hazard (#622 root cause, first-token no-sync): on a
        # mixed-arch rig ranks can read DIFFERENT first tokens and so
        # DISAGREE whether a request hit EOS -- one rank believes the
        # request FINISHED (freed its rows), peers still hold it parked.
        # The flip's replicated-live-set assumption is then false. This
        # test pins the required behavior: the flip REFUSES, and never
        # commits a mixed-membership layout silently.
        #
        # #656 C22: the refusal is now the pre-move frame ballot rather
        # than a payload error at the seam -- see the sibling test above
        # for why that is a strengthening and not a relaxation.
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
        #
        # #656 C22-d: the agreement is DISARMED here for the same reason as
        # the arm above -- this test measures what happens when nothing
        # reconciles the set, and it has to keep being able to go red. With
        # the agreement armed the group frames the union and cuts over; the
        # EOS disagreement itself is untouched by either behaviour, since it
        # exists before the flip and survives it.
        live_per_rank = [live, live, live_rank2]
        tp_before = _clone_pools(tp_pools)
        runtimes = self._runtimes_with_per_rank_live(live_per_rank, pp_views, tp_views)
        for rt in runtimes:
            rt._agree_live_slots = lambda slots, ballot: (slots, "")
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        self.assertEqual(
            [e for e in exceptions if e is not None],
            [],
            "a membership disagreement must be refused, not raised: raising "
            "at the seam takes the instance down",
        )
        # THE BINDING ASSERTIONS: detected on EVERY rank (the verdict is
        # the reduced one, so the refusal cannot be partial), no hang, and
        # -- stronger than the version this replaces -- not one byte of the
        # mixed-membership layout was committed anywhere.
        for r, rt in enumerate(runtimes):
            self.assertEqual(rt.frame_aborts, 1, f"rank {r}")
            self.assertEqual(rt.completed, 0, f"rank {r}")
        self.assertTrue(
            _pools_equal(tp_pools, tp_before),
            "a mixed-membership layout was scattered into the TP pools",
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
        runtimes = self._runtimes_with_per_rank_live(live_per_rank, pp_views, tp_views)
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
            threading.Thread(target=_worker, args=(r,), daemon=True) for r in range(n)
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


class _FakeClock:
    """Injectable monotonic clock. Tests advance time explicitly rather
    than sleeping, so the deadline is exercised in milliseconds."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class TestParkDeadline(CustomTestCase):
    """An armed flip withholds new work so the in-flight state drains; the
    flip can then interpose BETWEEN a request's prefill and its decode
    instead of only after every stream has finished. That park must be
    BOUNDED: a rank that never reaches quiescence would otherwise hold its
    requests forever.

    The contract under test, in the words of the requirement: on deadline
    expiry the FLIP aborts loudly; the user request is never aborted."""

    def _runtime(self, *, ready_seq, clock, deadline, channel_calls, rank=0):
        _, live, _, pp_views, _, tp_views = _make_layout_pools(
            MAP_625, VEC, 200, seed=41
        )

        def _min(vals):
            channel_calls.append(list(vals))
            return list(vals)

        def _ready():
            # A never-finishing stream: quiescence never arrives.
            return ready_seq.pop(0) if ready_seq else False

        return PhaseFlipRuntime(
            n_ranks=len(VEC),
            rank=rank,
            layer_map=MAP_625,
            n_layers=N_LAYERS,
            tp_vector=VEC,
            boot_phase=PHASE_PP,
            consensus_interval=2,
            park_deadline_s=deadline,
            collective_min=_min,
            exchange=lambda peers, payloads: {},
            pp_pool_view=pp_views[0],
            tp_pool_view=tp_views[0],
            live_slots_fn=lambda: live,
            ready_fn=_ready,
            cutover_fn=lambda d: None,
            clock=clock,
        )

    def test_never_quiescent_stream_abandons_the_flip_loudly(self):
        """THE falsifier: a synthetic never-finishing stream.

        RED before the deadline existed: the armed rank returned None
        every round forever, never entering the channel, and the requests
        it had parked never resumed."""
        clock = _FakeClock()
        calls = []
        rt = self._runtime(ready_seq=[], clock=clock, deadline=5.0, channel_calls=calls)
        ok, _ = rt.arm(PP_TO_TP, source="test")
        self.assertTrue(ok)

        # Inside the deadline: parked, silent, no collective at all.
        for _ in range(10):
            self.assertIsNone(rt.on_round(require_armed_and_parked=True))
        self.assertEqual(calls, [], "an unparked rank must not enter early")
        self.assertEqual(rt.pending, PP_TO_TP, "must stay armed inside the deadline")

        # Past the deadline: it enters ONCE, carrying `expired`, and the
        # group-agreed answer is to abandon.
        clock.advance(5.0)
        with self.assertLogs(
            "sglang.srt.managers.phase_flip_runtime", level="ERROR"
        ) as log:
            self.assertIsNone(rt.on_round(require_armed_and_parked=True))
        self.assertEqual(len(calls), 1, "expired rank must enter the consensus once")
        self.assertIn("FLIP ABANDONED", "\n".join(log.output))
        self.assertIsNone(rt.pending, "an abandoned flip must disarm")
        self.assertEqual(rt.park_deadline_aborts, 1)

    def test_abandon_does_not_raise_so_requests_survive(self):
        """The flip is optional; the parked requests are not.

        A raise here would climb into the event loop and take the instance
        down -- killing exactly the requests the deadline protects."""
        clock = _FakeClock()
        calls = []
        rt = self._runtime(ready_seq=[], clock=clock, deadline=1.0, channel_calls=calls)
        rt.arm(PP_TO_TP, source="test")
        clock.advance(2.0)
        # No exception, and serving continues in the SAME phase.
        self.assertIsNone(rt.on_round(require_armed_and_parked=True))
        self.assertEqual(rt.phase, PHASE_PP)
        self.assertEqual(rt.epoch, 0, "no flip happened")
        self.assertEqual(rt.completed, 0)

    def test_re_arming_after_abandon_restarts_the_clock(self):
        clock = _FakeClock()
        calls = []
        rt = self._runtime(ready_seq=[], clock=clock, deadline=1.0, channel_calls=calls)
        rt.arm(PP_TO_TP, source="test")
        clock.advance(2.0)
        rt.on_round(require_armed_and_parked=True)
        self.assertIsNone(rt.pending)

        rt.arm(PP_TO_TP, source="retry")
        self.assertEqual(rt.pending, PP_TO_TP)
        before = len(calls)
        for _ in range(5):
            rt.on_round(require_armed_and_parked=True)
        self.assertEqual(len(calls), before, "the re-armed clock must start from zero")

    def test_parked_rank_is_unaffected_by_the_deadline(self):
        """The healthy path must not change: parked ranks enter as before."""
        clock = _FakeClock()
        calls = []
        rt = self._runtime(
            ready_seq=[True] * 8, clock=clock, deadline=1.0, channel_calls=calls
        )
        rt.arm(PP_TO_TP, source="test")
        clock.advance(99.0)  # long past the deadline, but it PARKED
        try:
            rt.on_round(require_armed_and_parked=True)
        except Exception:
            pass  # the commit path runs further than this stub supports
        self.assertTrue(calls, "a parked rank must still enter the consensus")
        self.assertEqual(rt.park_deadline_aborts, 0, "a parked rank must never abandon")

    def test_zero_deadline_restores_the_unbounded_wait(self):
        clock = _FakeClock()
        calls = []
        rt = self._runtime(ready_seq=[], clock=clock, deadline=0.0, channel_calls=calls)
        rt.arm(PP_TO_TP, source="test")
        clock.advance(10_000.0)
        for _ in range(10):
            self.assertIsNone(rt.on_round(require_armed_and_parked=True))
        self.assertEqual(calls, [])
        self.assertEqual(rt.pending, PP_TO_TP)

    def test_expired_field_rides_the_consensus_payload(self):
        """Group-agreed, not rank-local: the flag must be IN the payload.

        A rank-local abandon would disarm one rank against still-armed
        peers -- the same rank-local-state-feeds-collective shape this
        family keeps producing."""
        clock = _FakeClock()
        calls = []
        rt = self._runtime(ready_seq=[], clock=clock, deadline=1.0, channel_calls=calls)
        rt.arm(PP_TO_TP, source="test")
        clock.advance(2.0)
        rt.on_round(require_armed_and_parked=True)
        self.assertEqual(len(calls), 1)
        payload = calls[0]
        # _encode packs (v, -v) pairs; expired is field index 2.
        self.assertEqual(payload[4], 1, "expired must be encoded in the payload")
        self.assertEqual(payload[5], -1)
        self.assertEqual(payload[0], 1, "armed")
        self.assertEqual(payload[2], 0, "ready")


class TestEmptyLiveSetFlip(CustomTestCase):
    """Flipping with NOTHING live must work, and must move nothing.

    This is not an exotic case: it is an idle server, and it is exactly
    what a caller reaches when it flushes the prefix cache to make room
    for the flip. It nevertheless killed every rank -- the byte view
    inferred its row width with view(n, -1), which torch refuses at n == 0
    ("cannot reshape tensor of 0 elements", #631 boot 20). The plan layer
    already handled the empty set correctly (max row -1); only the byte
    view did not."""

    def test_read_rows_of_no_rows_is_an_empty_payload(self):
        k = [torch.zeros(8, HEADS, DIM, dtype=torch.bfloat16)]
        v = [torch.zeros(8, HEADS, DIM, dtype=torch.bfloat16)]
        view = KvPoolView(k, v)
        empty = view.read_rows(0, torch.empty(0, dtype=torch.int64))
        self.assertEqual(tuple(empty.shape), (0, view.row_nbytes(0)))
        self.assertEqual(empty.dtype, torch.uint8)
        # And the width still agrees with a non-empty read, which is the
        # property the exchange's size check depends on.
        one = view.read_rows(0, torch.tensor([0], dtype=torch.int64))
        self.assertEqual(one.shape[1], empty.shape[1])

    def test_write_rows_accepts_the_empty_payload_and_changes_nothing(self):
        k = [torch.ones(8, HEADS, DIM, dtype=torch.bfloat16)]
        v = [torch.ones(8, HEADS, DIM, dtype=torch.bfloat16)]
        view = KvPoolView(k, v)
        before = k[0].clone()
        view.write_rows(
            0,
            torch.empty(0, dtype=torch.int64),
            torch.empty((0, view.row_nbytes(0)), dtype=torch.uint8),
        )
        self.assertTrue(torch.equal(k[0], before))

    def test_full_flip_with_an_empty_live_set(self):
        """End to end on the real runtime: an empty flip COMMITS."""
        _, _, _, pp_views, _, tp_views = _make_layout_pools(MAP_625, VEC, 64, seed=77)
        live = torch.empty(0, dtype=torch.int64)
        runtimes, _ = _build_runtimes(pp_views, tp_views, live)
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        for r, e in enumerate(exceptions):
            self.assertIsNone(e, f"rank {r} raised on an empty flip: {e}")
        for r, rt in enumerate(runtimes):
            self.assertEqual(rt.completed, 1, f"rank {r} did not complete")
            self.assertEqual(rt.phase, PHASE_TP, f"rank {r} did not change phase")
            self.assertEqual(rt.fit_aborts, 0)


def _make_aliased_layout_pools(layer_map, vec, num_slots, seed=7):
    """Same scenario as :func:`_make_layout_pools`, but each rank's PP and
    TP buffers are VIEWS INTO ONE per-rank arena -- the shape the shared
    cross-phase arena has, where the two layouts' bytes overlap because
    only one of them is physically backed at a time.

    The overlap is deliberate and adversarial: PP layer i and TP layer j
    are carved from the same flat storage at different strides, so any
    destination write that happens before a source read is READ has a real
    chance of clobbering a row still owed to the transition. With disjoint
    pools that ordering is unobservable, which is exactly why the hazard
    survived until sharing was attempted.
    """
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
    tp_rows_needed = 1
    for r in range(n_ranks):
        rr = rows_of(live[owner == r], vec, r)
        if rr.numel():
            tp_rows_needed = max(tp_rows_needed, int(rr.max().item()) + 1)

    arenas, pp_pools, pp_views, tp_pools, tp_views = [], [], [], [], []
    for r in range(n_ranks):
        # One arena per rank, big enough for whichever layout is larger --
        # max(PP, TP), never their sum. That IS the change under test.
        pp_rows_total = num_slots * len(layer_map[r])
        tp_rows_total = tp_rows_needed * N_LAYERS
        arena_rows = max(pp_rows_total, tp_rows_total)
        arena_k = torch.zeros(arena_rows, HEADS, DIM, dtype=torch.bfloat16)
        arena_v = torch.zeros(arena_rows, HEADS, DIM, dtype=torch.bfloat16)
        arenas.append((arena_k, arena_v))

        k_bufs = [
            arena_k[i * num_slots : (i + 1) * num_slots]
            for i in range(len(layer_map[r]))
        ]
        v_bufs = [
            arena_v[i * num_slots : (i + 1) * num_slots]
            for i in range(len(layer_map[r]))
        ]
        for i, f in enumerate(layer_map[r]):
            k_bufs[i][live] = ref_k[f][live]
            v_bufs[i][live] = ref_v[f][live]
        pp_pools.append((k_bufs, v_bufs))
        pp_views.append(KvPoolView(k_bufs, v_bufs))

        tk = [
            arena_k[i * tp_rows_needed : (i + 1) * tp_rows_needed]
            for i in range(N_LAYERS)
        ]
        tv = [
            arena_v[i * tp_rows_needed : (i + 1) * tp_rows_needed]
            for i in range(N_LAYERS)
        ]
        tp_pools.append((tk, tv))
        tp_views.append(KvPoolView(tk, tv))
    return (ref_k, ref_v), live, pp_pools, pp_views, tp_pools, tp_views


class TestSharedArenaReadsPrecedeWrites(CustomTestCase):
    """Falsifier for the cross-phase shared arena (#631 capacity follow-up).

    Both phases' KV pools are resident today, so the pool is roughly half
    what one layout could have. The fix is one arena per rank sized
    max(PP, TP) with mutually exclusive backing -- and it is only correct
    if the flip reads every source row before writing any destination row.

    This pins that ordering by making the two layouts alias. It fails on
    the pre-fix local leg, which read and wrote per layer in one loop.
    """

    def test_pp_to_tp_is_byte_exact_with_aliased_pools(self):
        ref, live, _pp_pools, pp_views, tp_pools, tp_views = _make_aliased_layout_pools(
            MAP_625, VEC, 300
        )
        runtimes, _cutovers = _build_runtimes(pp_views, tp_views, live)
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        self.assertEqual([e for e in exceptions if e], [])
        ok, msg = _check_tp_layout(tp_pools, ref, live, VEC)
        self.assertTrue(ok, f"aliased-arena flip corrupted rows: {msg}")

    def test_aliased_result_matches_the_disjoint_reference(self):
        """Byte-identity against the same flip run with disjoint pools --
        sharing must change memory economics, never a single output byte."""
        ref_a, live_a, _, pp_a, tp_pools_a, tp_a = _make_aliased_layout_pools(
            MAP_625, VEC, 300, seed=11
        )
        rt_a, _ = _build_runtimes(pp_a, tp_a, live_a)
        self.assertEqual(
            [e for e in _run_ranks(3, runtimes=rt_a, directions=[PP_TO_TP] * 3) if e],
            [],
        )

        ref_b, live_b, _, pp_b, tp_pools_b, tp_b = _make_layout_pools(
            MAP_625, VEC, 300, seed=11
        )
        rt_b, _ = _build_runtimes(pp_b, tp_b, live_b)
        self.assertEqual(
            [e for e in _run_ranks(3, runtimes=rt_b, directions=[PP_TO_TP] * 3) if e],
            [],
        )

        self.assertTrue(torch.equal(live_a, live_b))
        owner = owner_of(live_a, VEC)
        for r in range(3):
            rows = rows_of(live_a[owner == r], VEC, r)
            for f in range(N_LAYERS):
                self.assertTrue(
                    torch.equal(tp_pools_a[r][0][f][rows], tp_pools_b[r][0][f][rows]),
                    f"rank {r} ordinal {f} K differs between aliased and disjoint",
                )
                self.assertTrue(
                    torch.equal(tp_pools_a[r][1][f][rows], tp_pools_b[r][1][f][rows]),
                    f"rank {r} ordinal {f} V differs between aliased and disjoint",
                )


class TestSeamWavesAreByteIdentical(CustomTestCase):
    """#631: waving the seam changes memory economics, never a byte.

    The move is split into layer WAVES so that only one wave's payload is
    staged at a time -- the fix for the one-request livelock, where
    staging tracked the resident live set and a long enough request could
    never be afforded (HANDOFF_666). The wire format changes shape (one
    checksummed payload per peer PER WAVE instead of one for the whole
    plan), so the destination bytes are the thing that must not move.

    Run the SAME flip at one wave and at the map's default, on
    independently built but identically seeded pools, and compare every
    destination row. A single wave is the pre-wave code path, so this is
    also the A/B that keeps the wave count a one-variable change.
    """

    def _run_at(self, waves):
        ref, live, _pp_pools, pp_views, tp_pools, tp_views = _make_layout_pools(
            MAP_625, VEC, 300, seed=23
        )
        runtimes, _cutovers = _build_runtimes(pp_views, tp_views, live)
        for rt in runtimes:
            rt._n_waves = waves
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        self.assertEqual([e for e in exceptions if e], [])
        self.assertEqual([rt.last_stats["seam_waves"] for rt in runtimes], [waves] * 3)
        return ref, live, tp_pools

    def test_one_wave_and_the_default_split_agree_byte_for_byte(self):
        _ref_a, live_a, tp_a = self._run_at(1)
        _ref_b, live_b, tp_b = self._run_at(4)
        self.assertTrue(torch.equal(live_a, live_b))
        owner = owner_of(live_a, VEC)
        for r in range(3):
            rows = rows_of(live_a[owner == r], VEC, r)
            for f in range(N_LAYERS):
                self.assertTrue(
                    torch.equal(tp_a[r][0][f][rows], tp_b[r][0][f][rows]),
                    f"rank {r} ordinal {f} K differs between 1 and 4 waves",
                )
                self.assertTrue(
                    torch.equal(tp_a[r][1][f][rows], tp_b[r][1][f][rows]),
                    f"rank {r} ordinal {f} V differs between 1 and 4 waves",
                )

    def test_the_waved_run_really_did_land_the_reference_rows(self):
        """Equality between two runs proves nothing if both are wrong."""
        ref, live, tp_pools = self._run_at(4)
        ok, msg = _check_tp_layout(tp_pools, ref, live, VEC)
        self.assertTrue(ok, f"waved flip corrupted rows: {msg}")

    def test_the_default_wave_count_is_one_layer_per_wave(self):
        """#631 2.1b: the smallest-stage cap (4 for MAP_625) is lifted.

        Release-first needed every rank to own a layer in every wave, so
        the count stopped at the smallest stage. Restore-first budgets the
        overlap across the whole prefix instead, leaving one layer per wave
        as the only bound. ``default_wave_count`` still returns the old cap
        and is still the right answer for the aliased single-wave path --
        what changed is which of the two ``_flip_waves`` asks for.
        """
        ref, live, _pp, pp_views, tp_pools, tp_views = _make_layout_pools(
            MAP_625, VEC, 300, seed=23
        )
        runtimes, _ = _build_runtimes(pp_views, tp_views, live)
        waves = runtimes[0]._flip_waves(PP_TO_TP)
        self.assertEqual(len(waves), N_LAYERS)
        self.assertEqual(sorted(f for w in waves for f in w), list(range(N_LAYERS)))
        self.assertEqual(default_wave_count(MAP_625), 4)


class TestPreWriteSeamOrdering(CustomTestCase):
    """The read/write seam is where cross-phase KV backing may be swapped.

    Swapping physical pages between the two layouts is only safe at an
    instant where the SOURCE pool has been fully drained and the
    DESTINATION pool has not been touched. ``pre_write_fns`` exists to be
    that instant, so this pins it: on EVERY rank the hook must fire after
    that rank's last source read and before its first destination write.
    If it ever drifts to either side, a swap wired to it reads or writes
    unmapped memory.

    Ordering is checked per rank: the ranks run concurrently, so a global
    event order would interleave and prove nothing.
    """

    def test_hook_fires_between_last_read_and_first_write(self):
        ref, live, _pp_pools, pp_views, tp_pools, tp_views = _make_layout_pools(
            MAP_625, VEC, 300
        )
        events = {r: [] for r in range(len(VEC))}

        class _Recorder:
            def __init__(self, inner, tag, rank):
                self._inner = inner
                self._tag = tag
                self._rank = rank

            def read_rows(self, *a, **kw):
                events[self._rank].append(f"read:{self._tag}")
                return self._inner.read_rows(*a, **kw)

            def write_rows(self, *a, **kw):
                events[self._rank].append(f"write:{self._tag}")
                return self._inner.write_rows(*a, **kw)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        rec_pp = [_Recorder(v, "pp", r) for r, v in enumerate(pp_views)]
        rec_tp = [_Recorder(v, "tp", r) for r, v in enumerate(tp_views)]

        def _fns_for(rank):
            return (lambda direction, rank=rank: events[rank].append("SWAP"),)

        runtimes, _ = _build_runtimes(rec_pp, rec_tp, live, pre_write_fns_for=_fns_for)
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        self.assertEqual([e for e in exceptions if e], [])

        for r, log in events.items():
            self.assertIn("SWAP", log, f"rank {r} never reached the seam: {log}")
            seam = log.index("SWAP")
            self.assertEqual(
                [e for e in log[seam + 1 :] if e.startswith("read:")],
                [],
                f"rank {r}: a source read happened AFTER the seam: {log}",
            )
            self.assertEqual(
                [e for e in log[:seam] if e.startswith("write:")],
                [],
                f"rank {r}: a destination write happened BEFORE the seam: {log}",
            )

        ok, msg = _check_tp_layout(tp_pools, ref, live, VEC)
        self.assertTrue(ok, msg)


class _RecordingSeamSwap:
    """A ``WavedBackingSwap`` stand-in that records the seam's ORDER.

    ``_seam_swap`` picks the hook out of ``pre_write_fns`` by duck-typing on
    ``release_wave``, so a recorder carrying that surface drives the REAL
    wave loop. That is the whole reason this class exists:
    ``TestPreWriteSeamOrdering`` passes a plain callable, which takes the
    ``swap is None`` branch, so it can observe the seam's POSITION but never
    the order of release/reclaim/restore INSIDE it.

    Every call is logged unconditionally -- the real swap's "no layers of
    mine in this wave" early return is internal to it, and modelling that
    here would hide waves from the sequence assertions below.
    """

    is_swappable = True

    def __init__(self, log):
        self._log = log

    def release_wave(self, direction, wave):
        self._log.append(("release", tuple(int(f) for f in wave)))

    def restore_wave(self, direction, wave):
        self._log.append(("restore", tuple(int(f) for f in wave)))

    def reclaim_between(self, direction):
        self._log.append(("reclaim", None))

    # -- #631 2.1 span surface. Present so the streamed path engages; the
    # recorder still swaps no real backing, which is deliberate -- it lets
    # the byte-identity assertion isolate the WRITE ORDER from the backing.
    def is_span_swappable(self, direction):
        return True

    def restore_wave_span(self, direction, wave, lo, hi):
        self._log.append(("restore_span", (int(lo), int(hi))))

    def release_wave_span(self, direction, wave, lo, hi):
        self._log.append(("release_span", (int(lo), int(hi))))

    def finalize_wave(self, direction, wave):
        self._log.append(("finalize", tuple(int(f) for f in wave)))


class TestWavedSeamOrdering(CustomTestCase):
    """The waved seam's per-wave order: reclaim -> restore -> release.

    #631 section 2.1b. Two DIFFERENT hazards are pinned here and they pull
    in opposite directions, which is why both assertions have to exist:

    * RESTORE BEFORE RELEASE, per wave. Under release-first a wave's
      destination pages are committed only after its source pages are gone,
      so the staging peak carries a full wave of drift and the seam's slope
      is ~4.5 MiB per 1000 live slots per wave. Restore-first budgets the
      overlap explicitly, which is what lets the wave count rise to
      ``n_layers`` and drops the slope to ~1.1.

    * RECLAIM AHEAD OF THE FIRST RESTORE. The restore is the allocation that
      can fail INSIDE the no-return region; it OOM'd on metal on 2026-08-09
      and took the instance down. Under release-first the reclaim naturally
      sat at the memory trough. Restore-first moves the destination commit
      to the PEAK with the source still mapped, so the reclaim must move
      AHEAD of it rather than following it. An implementation that reorders
      the pair to restore -> reclaim -> release re-opens that crash and
      must fail here, not on the rig.

    The whole-pool ``WavedBackingSwap.__call__`` path is NOT covered by this
    and must KEEP release-first: it holds BOTH layouts for the width of the
    swap, which is the residency the waved seam exists to remove. Its pins
    live in ``SeamOrderingTest``
    (test/registered/unit/managers/test_phase_flip_spill_depth_631.py).
    """

    def _run_and_log(self):
        ref, live, _pp_pools, pp_views, tp_pools, tp_views = _make_layout_pools(
            MAP_625, VEC, 300
        )
        logs = {r: [] for r in range(len(VEC))}
        runtimes, _ = _build_runtimes(
            pp_views,
            tp_views,
            live,
            pre_write_fns_for=lambda r: (_RecordingSeamSwap(logs[r]),),
        )
        # WHOLE-WAVE ARM, PINNED EXPLICITLY. This class asserts the order of
        # release/reclaim/restore INSIDE a whole-layer commit; the streamed
        # path (now the default block count) emits a different, equally
        # correct sequence, which TestStreamedSeamOrdering owns. Leaving the
        # arm implicit made these tests silently follow the default and fail
        # the moment it moved -- the arm under test must be stated, not
        # inherited.
        for rt in runtimes:
            rt._seam_row_blocks = 1
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        self.assertEqual([e for e in exceptions if e], [])
        return logs, tp_pools, ref, live

    def test_the_seam_is_waved_at_all(self):
        """Guard on the fixture: a single wave would make the rest vacuous."""
        logs, _tp_pools, _ref, _live = self._run_and_log()
        for r, log in logs.items():
            waves = [w for kind, w in log if kind == "restore"]
            self.assertGreater(
                len(waves), 1, f"rank {r}: seam ran unwaved, sequence proves nothing"
            )

    def test_each_wave_restores_the_destination_before_releasing_the_source(self):
        logs, _tp_pools, _ref, _live = self._run_and_log()
        for r, log in logs.items():
            seen = [e for e in log if e[0] in ("restore", "release")]
            for i in range(0, len(seen), 2):
                pair = seen[i : i + 2]
                self.assertEqual(
                    [k for k, _w in pair],
                    ["restore", "release"],
                    f"rank {r}: wave pair {i // 2} ran {[k for k, _w in pair]}, "
                    f"expected restore then release -- release-first is the "
                    f"4.5 MiB/1000-slot slope that caps the pool at ~438k "
                    f"(#631 section 2.1b). Full sequence: {log}",
                )
                self.assertEqual(
                    pair[0][1],
                    pair[1][1],
                    f"rank {r}: restore and release disagree about the wave; "
                    f"the pair must be the SAME wave's layers: {log}",
                )

    def test_reclaim_runs_once_and_ahead_of_the_first_restore(self):
        logs, _tp_pools, _ref, _live = self._run_and_log()
        for r, log in logs.items():
            kinds = [k for k, _w in log]
            self.assertEqual(
                kinds.count("reclaim"),
                1,
                f"rank {r}: reclaim is a once-per-flip rung, got "
                f"{kinds.count('reclaim')}: {log}",
            )
            self.assertLess(
                kinds.index("reclaim"),
                kinds.index("restore"),
                f"rank {r}: the reclaim must hand pages back BEFORE the first "
                f"destination commit -- that commit is the allocation which "
                f"OOM'd inside the no-return region on 2026-08-09. "
                f"Sequence: {log}",
            )

    def test_waves_do_not_interleave(self):
        """Wave j's release must precede wave j+1's restore.

        Otherwise two waves' destination pages are committed at once and the
        peak is two waves wide, which silently undoes the bound that the
        wave split exists to provide.
        """
        logs, _tp_pools, _ref, _live = self._run_and_log()
        for r, log in logs.items():
            order = [(k, w) for k, w in log if k in ("restore", "release")]
            for i in range(2, len(order), 2):
                self.assertEqual(
                    order[i - 1][0],
                    "release",
                    f"rank {r}: wave {i // 2} began before the previous wave "
                    f"released: {log}",
                )

    def test_aliased_pools_keep_release_first(self):
        """The alias gate, asserted on ORDER rather than on bytes.

        ``TestSharedArenaReadsPrecedeWrites`` covers the aliased path but
        checks byte identity only, so if this gate were wrong it would stay
        green wherever the corruption happened to be invisible to that
        fixture. It is asserted directly here because the failure is not a
        slow path or a lost optimisation: when the layouts overlay the same
        bytes, the destination's pages ARE the source's, so restoring and
        then releasing hands back the mapping just committed and leaves the
        destination unbacked.
        """
        _ref, live, _pp_pools, pp_views, _tp_pools, tp_views = (
            _make_aliased_layout_pools(MAP_625, VEC, 300)
        )
        logs = {r: [] for r in range(len(VEC))}
        runtimes, _ = _build_runtimes(
            pp_views,
            tp_views,
            live,
            pre_write_fns_for=lambda r: (_RecordingSeamSwap(logs[r]),),
        )
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        self.assertEqual([e for e in exceptions if e], [])
        for r, log in logs.items():
            kinds = [k for k, _w in log]
            self.assertEqual(
                kinds.count("restore"),
                1,
                f"rank {r}: the aliased seam must run as ONE wave: {log}",
            )
            self.assertLess(
                kinds.index("release"),
                kinds.index("restore"),
                f"rank {r}: aliased layouts must release the source BEFORE "
                f"restoring the destination -- they are the same pages. "
                f"Sequence: {log}",
            )

    def _run_blocked(self, blocks, seed=31):
        ref, live, _pp, pp_views, tp_pools, tp_views = _make_layout_pools(
            MAP_625, VEC, 300, seed=seed
        )
        logs = {r: [] for r in range(len(VEC))}
        runtimes, _ = _build_runtimes(
            pp_views,
            tp_views,
            live,
            pre_write_fns_for=lambda r: (_RecordingSeamSwap(logs[r]),),
        )
        for rt in runtimes:
            rt._seam_row_blocks = blocks
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        self.assertEqual([e for e in exceptions if e], [])
        return logs, tp_pools, ref, live

    def test_the_streamed_seam_writes_the_same_bytes_as_the_whole_wave_one(self):
        """#631 2.1: blocking changes WHEN pages are mapped, never a byte.

        The strongest available falsifier: the same flip, same seed, run at
        two block counts, compared tensor by tensor. A row-blocking bug
        that mis-slices the payload -- writing a peer's rows at the wrong
        offset -- lands here and essentially nowhere else, because every
        row still gets written exactly once and the counts still agree.
        """
        _l1, tp1, ref1, live1 = self._run_blocked(1)
        _l4, tp4, ref4, live4 = self._run_blocked(4)
        self.assertTrue(torch.equal(live1, live4))
        ok, msg = _check_tp_layout(tp4, ref4, live4, VEC)
        self.assertTrue(ok, f"streamed seam corrupted rows: {msg}")
        for r, ((k1, v1), (k4, v4)) in enumerate(zip(tp1, tp4)):
            for f in range(len(k1)):
                self.assertTrue(
                    torch.equal(k1[f], k4[f]),
                    f"rank {r} layer {f}: streamed K differs from whole-wave K",
                )
                self.assertTrue(
                    torch.equal(v1[f], v4[f]),
                    f"rank {r} layer {f}: streamed V differs from whole-wave V",
                )

    def test_each_block_restores_then_releases_and_the_wave_is_finalised(self):
        logs, _tp, _ref, _live = self._run_blocked(4)
        for r, log in logs.items():
            # One segment per wave, cut at its finalize.
            segments, cur = [], []
            for kind, payload in log:
                if kind == "reclaim":
                    continue
                cur.append(kind)
                if kind == "finalize":
                    segments.append(cur)
                    cur = []
            self.assertTrue(segments, f"rank {r}: streamed path never engaged")
            self.assertEqual(cur, [], f"rank {r}: a wave never reached finalize: {cur}")
            for w, seg in enumerate(segments):
                self.assertEqual(
                    seg,
                    ["restore_span", "release_span"] * 4 + ["finalize"],
                    f"rank {r} wave {w}: each block must restore then "
                    f"release, and finalize must close the wave AFTER the "
                    f"last block -- finalising early marks the pool "
                    f"resident while it is still partly unbacked. Got: {seg}",
                )

    def test_blocking_shrinks_the_commit_unit(self):
        """The point of the change, asserted as a count.

        Four blocks must produce four commit units per wave, not one. If
        the knob silently fell back to whole-wave commits the transient
        would be unchanged and every other test here would still pass.
        """
        for blocks in (2, 4):
            logs, _tp, _ref, _live = self._run_blocked(blocks)
            for r, log in logs.items():
                restores = [k for k, _w in log if k == "restore_span"]
                waves = [k for k, _w in log if k == "finalize"]
                self.assertEqual(
                    len(restores),
                    blocks * len(waves),
                    f"rank {r}: expected {blocks} commit units per wave",
                )

    def test_unsorted_rows_are_refused_rather_than_mis_sliced(self):
        """The can-fail proof for the ascending-rows assumption.

        Selecting scattered writes by contiguous row RANGE is only valid
        because the plan enumerates slots ascending. If that ever changes,
        the slice would pair the wrong payload with the wrong rows -- a
        silent KV corruption. It must raise instead.
        """
        rt = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
        rows = torch.tensor([5, 1, 9], dtype=torch.int64)
        data = torch.zeros(3, 4, dtype=torch.uint8)
        with self.assertRaises(KvReshardError):
            from types import SimpleNamespace as _NS

            rt._stream_wave(
                _RecordingSeamSwap([]),
                PP_TO_TP,
                (0,),
                _NS(num_rows=16),
                _NS(num_rows=16),
                [(0, rows, data)],
                4,
            )

    def test_reordering_the_seam_changes_no_byte(self):
        """Byte identity is the net under the whole reorder.

        Restore-first changes WHEN physical pages are mapped, never which
        bytes are read or written, so the flip's output must be unchanged.
        """
        logs, tp_pools, ref, live = self._run_and_log()
        self.assertTrue(all(logs.values()))
        ok, msg = _check_tp_layout(tp_pools, ref, live, VEC)
        self.assertTrue(ok, f"the reordered seam corrupted rows: {msg}")


# -- #631 section 2.1 PREREQUISITES (successor 27) ---------------------------
#
# The row-blocked seam shipped dark in ab3f3e6460 and could not have run on
# any boot this rig has taken. Three linked reasons, one test each below,
# plus the accounting that decides whether the shrink is ever CASHED.


class TestStreamedSeamPrerequisites(CustomTestCase):
    """The gate must answer for the ARENA, not for the method table.

    ``commit_span``/``decommit_span`` raise unless the arena was built with
    a commit chunk (``_require_chunk``), because ``cuMemUnmap`` only takes
    whole mappings and a monolithic per-buffer extent can therefore only be
    released all-or-nothing. ``SGLANG_FLIP_SEAM_CHUNK_MIB`` defaults to 0,
    so on a default boot the span API is present and non-functional -- and
    ``hasattr`` cannot tell those apart. The raise would land inside the
    flip's no-return region.
    """

    @staticmethod
    def _swap(supports):
        from types import SimpleNamespace as NS

        pool = NS(
            release_backing_span=lambda *a: 0,
            restore_backing_span=lambda *a: 0,
            supports_backing_spans=supports,
        )
        swap = WavedBackingSwap.__new__(WavedBackingSwap)
        swap._pp_pool = pool
        swap._tp_pool = pool
        return swap

    def test_span_swappable_is_false_without_a_commit_chunk(self):
        swap = self._swap(False)
        self.assertFalse(
            swap.is_span_swappable(PP_TO_TP),
            "is_span_swappable answered from hasattr, so a pool whose arena "
            "has no commit chunk green-lights a path that raises inside the "
            "seam's no-return region",
        )

    def test_span_swappable_is_true_with_one(self):
        self.assertTrue(self._swap(True).is_span_swappable(PP_TO_TP))


class TestSeamChunkRetentionDecoupled(CustomTestCase):
    """A commit chunk must be obtainable WITHOUT handle retention.

    Retention parks released handles in ``KvVmmArena._retained`` for reuse.
    That is per-ARENA, and the two layouts are two arenas: the PP arena
    parks the pages the TP arena needs, and the TP arena's
    ``_take_retained`` can never see them. Both layouts then stay resident
    and exclusive backing -- the whole reason this pool is VA-backed -- is
    defeated. Row-blocking NEEDS the chunk, so the two must come apart.
    """

    def test_chunk_alone_does_not_turn_retention_on(self):
        from sglang.srt.mem_cache.memory_pool import seam_chunk_and_retention

        chunk, retain = seam_chunk_and_retention(64, swappable=True, retain_env="")
        self.assertEqual(chunk, 64 << 20)
        self.assertFalse(
            retain,
            "retention is per-arena and the two layouts are two arenas, so "
            "parking the source's pages hides them from the destination",
        )

    def test_retention_is_still_reachable_on_purpose(self):
        from sglang.srt.mem_cache.memory_pool import seam_chunk_and_retention

        chunk, retain = seam_chunk_and_retention(64, swappable=True, retain_env="1")
        self.assertEqual(chunk, 64 << 20)
        self.assertTrue(retain)

    def test_no_chunk_means_no_chunk_and_no_retention(self):
        from sglang.srt.mem_cache.memory_pool import seam_chunk_and_retention

        self.assertEqual(
            seam_chunk_and_retention(0, swappable=True, retain_env="1"), (None, False)
        )
        self.assertEqual(
            seam_chunk_and_retention(64, swappable=False, retain_env=""), (None, False)
        )


class TestStreamedReleaseCoversBoundaryChunks(CustomTestCase):
    """The residual that grows with B and cancels the 1/B gain.

    ``decommit_span`` rounds INWARD (a chunk only partly inside the range
    still holds live rows). So for an interior boundary, block ``b``'s
    ``hi`` rounds below it and block ``b+1``'s ``lo`` rounds above it, and
    the chunk straddling it is released by NEITHER -- ``(B-1)`` chunks per
    buffer left mapped on the resting layout, growing with the block count.

    The fix costs nothing: release ``[0, hi_b)`` cumulatively. Every source
    read completes before the seam opens (``_execute`` reads the retained
    leg and drains the exchange first), so no source row is live at any
    point in this loop, and a wider release is always sound.
    """

    def test_release_spans_are_cumulative_from_row_zero(self):
        from types import SimpleNamespace as NS

        log = []
        swap = _RecordingSeamSwap(log)
        rt = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
        rows = torch.arange(8, dtype=torch.int64)
        data = torch.zeros(8, 4, dtype=torch.uint8)
        written = []
        dst = NS(
            num_rows=16,
            write_rows=lambda li, r, d: written.append((li, r.clone())),
        )
        rt._stream_wave(
            swap, PP_TO_TP, (0,), NS(num_rows=16), dst, [(0, rows, data)], 4
        )
        rel = [span for kind, span in log if kind == "release_span"]
        self.assertEqual(
            rel,
            [(0, 4), (0, 8), (0, 12), (0, 16)],
            "each block must release everything released so far, so the "
            "chunk straddling a block boundary -- which inward rounding "
            "leaves out of both neighbours -- is covered by the next block",
        )

    def test_restore_spans_stay_disjoint(self):
        """Only the RELEASE side widens.

        Commit rounds outward and is idempotent, so a disjoint restore
        already leaves no gap; widening it too would re-commit the whole
        destination on every block and put the layer-span transient
        straight back.
        """
        from types import SimpleNamespace as NS

        log = []
        rt = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
        rows = torch.arange(8, dtype=torch.int64)
        data = torch.zeros(8, 4, dtype=torch.uint8)
        dst = NS(num_rows=16, write_rows=lambda li, r, d: None)
        rt._stream_wave(
            _RecordingSeamSwap(log),
            PP_TO_TP,
            (0,),
            NS(num_rows=16),
            dst,
            [(0, rows, data)],
            4,
        )
        res = [span for kind, span in log if kind == "restore_span"]
        self.assertEqual(res, [(0, 4), (4, 8), (8, 12), (12, 16)])


class TestBlockedSeamAccounting(CustomTestCase):
    """The gate must PRICE the shrink, or the shrink is never cashed.

    ``_staging_bytes`` is what refuses a flip. Row-blocking makes the real
    peak smaller, but if ``_backing_slack_bytes`` keeps charging a whole
    layer span the reservation is unchanged, the pool stays capped exactly
    where it was, and the change measures as inert. The mirror error is
    worse: pricing blocks that the loop is not running under-reserves and
    the flip OOMs inside the no-return region.
    """

    @staticmethod
    def _rt(blocks, restore_first=True, span_swappable=True, chunk=0):
        from types import SimpleNamespace as NS

        rt = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
        rt._map = MAP_625
        rt._rank = 0
        rt._seam_row_blocks = blocks
        rt._seam_restore_first = restore_first
        rt._pools_alias = lambda: False
        rt._seam_backing_is_swappable = lambda: True
        rt._seam_swap = lambda: NS(
            is_swappable=True,
            is_span_swappable=lambda d: span_swappable,
            commit_chunk_bytes=lambda d: chunk,
        )
        return rt

    @staticmethod
    def _pools():
        from types import SimpleNamespace as NS

        # THE TWO SPANS MUST DIFFER OR THE TEST PROVES NOTHING. A PP layer
        # carries every head over the whole pool; a TP layer carries this
        # rank's head share (3 of 7 by VEC) over the whole pool. With equal
        # spans the wave's NET growth swamps the transient and every block
        # count prices the same -- which is a property of the fixture, not
        # of the accounting, and it is what the first version of this test
        # accidentally asserted.
        src = NS(num_layers=16, num_rows=1000, row_nbytes=lambda i: 7)
        dst = NS(num_layers=16, num_rows=1000, row_nbytes=lambda i: 3)
        return src, dst

    def _slack(self, rt, waves=None):
        src, dst = self._pools()
        if waves is None:
            waves = tuple((f,) for f in range(N_LAYERS))
        return rt._backing_slack_bytes(PP_TO_TP, src, dst, waves)

    def test_blocking_shrinks_the_reservation(self):
        one = self._slack(self._rt(1))
        four = self._slack(self._rt(4))
        self.assertLess(
            four,
            one,
            "the gate charged the whole layer span regardless of the block "
            "count, so a streamed seam would shrink the real peak and still "
            "be refused at exactly the old pool ceiling",
        )

    def test_one_block_is_byte_identical_to_the_unblocked_price(self):
        """Keeps the block count a ONE-VARIABLE A/B."""
        self.assertEqual(self._slack(self._rt(1)), self._slack(self._rt(0)))

    def test_the_price_never_undercuts_what_the_loop_will_do(self):
        """Blocks priced but not run is the dangerous direction.

        Release-first, aliased pools and an unchunked arena all take the
        whole-wave branch in ``_execute``. Charging the blocked price there
        under-reserves, and the gate's verdict is the last check before the
        no-return region.
        """
        unblocked = self._slack(self._rt(1))
        self.assertEqual(
            self._slack(self._rt(4, span_swappable=False)),
            unblocked,
            "an arena with no commit chunk cannot do span ops, so _execute "
            "takes the whole-wave branch and the price must follow it",
        )
        # Release-first has its own, correct, price; the only requirement is
        # that the block count cannot move it, because that branch never
        # streams.
        self.assertEqual(
            self._slack(self._rt(4, restore_first=False)),
            self._slack(self._rt(1, restore_first=False)),
        )

    def test_the_chunk_granularity_is_a_floor(self):
        """More blocks cannot commit less than one chunk per buffer.

        Without this the model claims a shrink the driver cannot deliver
        (``commit_span`` rounds OUTWARD to the chunk), which is an
        under-reservation.
        """
        fine = self._slack(self._rt(64, chunk=0))
        floored = self._slack(self._rt(64, chunk=200))
        self.assertGreater(
            floored, fine, "the chunk floor did not bind at a fine block count"
        )

    def test_blocking_converges_to_the_net_not_to_zero(self):
        """Honesty check on the limit.

        The transient shrinks as 1/B, but the wave's NET growth
        (commit minus release) does not -- it is layout arithmetic, not
        ordering. A model that drove the whole term to zero would be
        promising a free flip.
        """
        prices = [self._slack(self._rt(b)) for b in (1, 2, 4, 8, 16, 64)]
        self.assertEqual(prices, sorted(prices, reverse=True))
        self.assertGreater(prices[-1], 0)


class TestTheWrapperCannotDropTheSpanSurface(CustomTestCase):
    """The gap that made section 2.1 dead code on every boot.

    The object the flip holds is ``HybridLinearKVPool``, a wrapper that
    forwards the backing calls to its full-attention sub-pool. It forwarded
    ``release_backing``/``restore_backing`` and NOT their span variants, so
    the seam's capability probe looked straight past a capability the
    underlying pool had, and took the whole-wave branch on every flip
    without saying so. A capability probe a wrapper can drop is a
    capability that turns itself off, which is the worst kind: nothing
    fails, the feature is simply never exercised.

    Pinned as a SURFACE test rather than through a flip, because the flip
    is exactly what could not observe it.
    """

    SPAN_SURFACE = (
        "release_backing_span",
        "restore_backing_span",
        "supports_backing_spans",
        "backing_commit_chunk_bytes",
    )

    def test_the_hybrid_wrapper_forwards_every_span_member(self):
        from sglang.srt.mem_cache.memory_pool import (
            HybridLinearKVPool,
            MHATokenToKVPool,
        )

        for name in self.SPAN_SURFACE:
            self.assertTrue(
                hasattr(MHATokenToKVPool, name),
                f"{name} missing on the pool that owns the arena",
            )
            self.assertTrue(
                hasattr(HybridLinearKVPool, name),
                f"HybridLinearKVPool does not forward {name}; the seam holds "
                f"the wrapper, so an unforwarded member silently disables "
                f"the streamed path",
            )

    def test_forwarding_reaches_the_sub_pool(self):
        from types import SimpleNamespace as NS

        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        calls = []
        w = HybridLinearKVPool.__new__(HybridLinearKVPool)
        w.full_kv_pool = NS(
            release_backing_span=lambda l, lo, hi: calls.append(("rel", lo, hi)) or 7,
            restore_backing_span=lambda l, lo, hi: calls.append(("res", lo, hi)) or 9,
            supports_backing_spans=True,
            backing_commit_chunk_bytes=16 << 20,
        )
        self.assertEqual(w.release_backing_span([0], 1, 2), 7)
        self.assertEqual(w.restore_backing_span([0], 3, 4), 9)
        self.assertEqual(calls, [("rel", 1, 2), ("res", 3, 4)])
        self.assertTrue(w.supports_backing_spans)
        self.assertEqual(w.backing_commit_chunk_bytes, 16 << 20)

    def test_a_sub_pool_without_the_capability_answers_no(self):
        """Can-fail proof: the forward must not manufacture a yes."""
        from types import SimpleNamespace as NS

        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        w = HybridLinearKVPool.__new__(HybridLinearKVPool)
        w.full_kv_pool = NS()
        self.assertFalse(w.supports_backing_spans)
        self.assertEqual(w.backing_commit_chunk_bytes, 0)


class _RecordingFlush:
    """#787 SENDER-SIDE HALF pin: records, at the moment it fires, whether
    the runtime's flip was still pending -- i.e. whether the flush ran
    BEFORE local flip state was cleared.

    The runtime object does not exist yet when this stub is built (it is
    passed INTO the ``PhaseFlipRuntime`` constructor), so the reference is
    bound after construction via ``.runtime = rt`` rather than captured at
    stub-creation time.
    """

    def __init__(self):
        self.runtime = None
        self.pending_at_call = []

    def __call__(self):
        rt = self.runtime
        self.pending_at_call.append(None if rt is None else rt.pending)


class TestAbandonFlushesPendingSendsBeforeClearing787(CustomTestCase):
    """#787 sender-side half, pinned directly (no drain/settle window in
    sight -- that receiver-side half is covered elsewhere, this class only
    watches the abandon paths' ordering promise).

    ``_abandon_no_quorum`` and ``_abandon_unjoined_flip`` both claim, in a
    code comment, that they flush/count pending CHAN_DICT sends BEFORE
    clearing local flip state (``self._pending = None``). A comment is not
    a pin (#505b): this class constructs a real ``PhaseFlipRuntime``, wires
    a recording ``flush_pending_sends_fn`` into it, drives each abandon path
    directly, and asserts both that the hook fired and that the flip was
    still pending at the moment it fired.
    """

    def _armed_runtime(self, flush):
        _, live, _, pp_views, _, tp_views = _make_layout_pools(
            MAP_625, VEC, 100, seed=61
        )
        rt = PhaseFlipRuntime(
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
            flush_pending_sends_fn=flush,
        )
        flush.runtime = rt
        ok, msg = rt.arm(PP_TO_TP, source="test")
        self.assertTrue(ok, msg)
        self.assertEqual(rt.pending, PP_TO_TP, "precondition: the flip is armed")
        return rt

    def test_no_quorum_abandon_flushes_before_clearing(self):
        flush = _RecordingFlush()
        rt = self._armed_runtime(flush)
        rt._abandon_no_quorum(epoch=0, missing=(1, 2), waited=3.5)

        self.assertEqual(len(flush.pending_at_call), 1, "hook must fire exactly once")
        self.assertEqual(
            flush.pending_at_call[0],
            PP_TO_TP,
            "the flush ran while the flip was still pending -- i.e. BEFORE "
            "the abandon cleared local state, not after",
        )
        self.assertIsNone(rt.pending, "the abandon must still clear afterwards")

    def test_unjoined_abandon_flushes_before_clearing(self):
        flush = _RecordingFlush()
        rt = self._armed_runtime(flush)
        rt._abandon_unjoined_flip(why="synthetic join timeout")

        self.assertEqual(len(flush.pending_at_call), 1, "hook must fire exactly once")
        self.assertEqual(
            flush.pending_at_call[0],
            PP_TO_TP,
            "the flush ran while the flip was still pending -- i.e. BEFORE "
            "the abandon cleared local state, not after",
        )
        self.assertIsNone(rt.pending, "the abandon must still clear afterwards")

    def test_missing_hook_is_tolerated(self):
        """Contrast: an unset hook (the default) must not be required --
        callers that never wired one keep working exactly as before."""
        _, live, _, pp_views, _, tp_views = _make_layout_pools(
            MAP_625, VEC, 100, seed=62
        )
        rt = PhaseFlipRuntime(
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
        ok, msg = rt.arm(PP_TO_TP, source="test")
        self.assertTrue(ok, msg)
        rt._abandon_no_quorum(epoch=0, missing=(1,), waited=1.0)
        self.assertIsNone(rt.pending)
